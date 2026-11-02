import React, { useEffect, useState } from "react";
import { Accordion } from "@base-ui/react/accordion";
import {
  Button as BaseButton,
  type ButtonProps as BaseButtonProps,
} from "@base-ui/react/button";
import { Input as BaseInput } from "@base-ui/react/input";
import { Select } from "@base-ui/react/select";
import { Switch } from "@base-ui/react/switch";
import { Tabs } from "@base-ui/react/tabs";
import { Tooltip } from "@base-ui/react/tooltip";
import { IconCheck, IconChevronDown } from "../ui";

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  return (
    <Tooltip.Provider delay={260} closeDelay={70}>
      {children}
    </Tooltip.Provider>
  );
}

export interface ActionButtonProps extends Omit<BaseButtonProps, "className"> {
  className?: string;
  variant?: "primary" | "secondary" | "ghost" | "danger" | "link" | "on-dark";
  size?: "md" | "sm" | "icon";
  tooltip?: string;
  tooltipSide?: "top" | "bottom" | "left" | "right";
}

export function ActionButton({
  className,
  variant = "secondary",
  size = "md",
  type = "button",
  tooltip,
  tooltipSide = "top",
  disabled,
  children,
  ...rest
}: ActionButtonProps) {
  const classes = [
    "ui-btn",
    `ui-btn-${variant}`,
    `ui-btn-${size}`,
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const buttonElement = (
    <BaseButton
      type={type}
      disabled={disabled}
      className={classes}
      {...rest}
    >
      {children}
    </BaseButton>
  );

  if (!tooltip) return buttonElement;

  return (
    <Tooltip.Root>
      <Tooltip.Trigger disabled={Boolean(disabled)} render={buttonElement} />
      <Tooltip.Portal>
        <Tooltip.Positioner
          className="ui-tooltip-positioner"
          side={tooltipSide}
          sideOffset={7}
        >
          <Tooltip.Popup className="ui-tooltip-popup">{tooltip}</Tooltip.Popup>
        </Tooltip.Positioner>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function TextField({
  label,
  value,
  set,
  type = "text",
  placeholder = "",
  mono = false,
}: {
  label: string;
  value: string | number;
  set: (value: string) => void;
  type?: string;
  placeholder?: string;
  mono?: boolean;
}) {
  const id = React.useId();
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{label}</span>
      <BaseInput
        id={id}
        className={mono ? "ui-input mono-input" : "ui-input"}
        type={type}
        value={value}
        placeholder={placeholder}
        autoComplete="off"
        onValueChange={(next) => set(next)}
      />
    </label>
  );
}

export interface SelectOption {
  value: string;
  label: string;
}

export function withCurrentOption(
  options: readonly SelectOption[],
  currentValue: string,
  emptyOptionLabel?: string,
): SelectOption[] {
  const seen = new Set<string>();
  const list: SelectOption[] = [];

  if (emptyOptionLabel !== undefined) {
    seen.add("");
    list.push({ value: "", label: emptyOptionLabel });
  }

  for (const item of options) {
    if (!seen.has(item.value)) {
      seen.add(item.value);
      list.push(item);
    }
  }

  if (!seen.has(currentValue)) {
    seen.add(currentValue);
    list.push({
      value: currentValue,
      label: currentValue === "" ? emptyOptionLabel ?? "—" : currentValue,
    });
  }

  return list;
}

export interface SelectFieldProps {
  label?: string;
  ariaLabel?: string;
  srOnlyLabel?: boolean;
  inlineLabel?: boolean;
  compact?: boolean;
  mono?: boolean;
  disabled?: boolean;
  className?: string;
  value: string;
  options: readonly SelectOption[];
  onChange: (value: string) => void;
}

export function SelectField({
  label,
  ariaLabel,
  srOnlyLabel = false,
  inlineLabel = false,
  compact = false,
  mono = false,
  disabled = false,
  className,
  value,
  options,
  onChange,
}: SelectFieldProps) {
  const resolved = withCurrentOption(options, value);
  const wrapperClass = [
    inlineLabel ? "ui-select-inline" : "field",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  const triggerClass = [
    "ui-select-trigger",
    compact ? "compact" : "",
    mono ? "mono-input" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={wrapperClass}>
      <Select.Root<string>
        items={resolved}
        value={value}
        onValueChange={(next) => onChange(next ?? "")}
        disabled={disabled}
      >
        {label && (
          <Select.Label
            className={
              srOnlyLabel
                ? "sr-only"
                : inlineLabel
                  ? "ui-select-inline-label"
                  : "field-label"
            }
          >
            {label}
          </Select.Label>
        )}
        <Select.Trigger
          type="button"
          className={triggerClass}
          aria-label={ariaLabel}
        >
          <Select.Value className="ui-select-value">
            {(selected: string | null) => {
              const current = selected ?? "";
              const found = resolved.find((item) => item.value === current);
              return found ? found.label : current || "—";
            }}
          </Select.Value>
          <Select.Icon className="ui-select-icon">
            <IconChevronDown size={14} />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Positioner
            className="ui-select-positioner"
            sideOffset={6}
            alignItemWithTrigger={false}
          >
            <Select.Popup className="ui-select-popup">
              <Select.List className="ui-select-list">
                {resolved.map((opt) => (
                  <Select.Item
                    key={opt.value === "" ? "empty" : `v:${opt.value}`}
                    value={opt.value}
                    label={opt.label}
                    className={
                      mono ? "ui-select-item mono-input" : "ui-select-item"
                    }
                  >
                    <Select.ItemText className="ui-select-item-text">
                      {opt.label}
                    </Select.ItemText>
                    <Select.ItemIndicator className="ui-select-item-check">
                      <IconCheck size={14} />
                    </Select.ItemIndicator>
                  </Select.Item>
                ))}
              </Select.List>
            </Select.Popup>
          </Select.Positioner>
        </Select.Portal>
      </Select.Root>
    </div>
  );
}

export function SwitchField({
  title,
  description,
  checked,
  onChange,
  disabled = false,
}: {
  title: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  const switchId = React.useId();
  const labelId = React.useId();
  const descId = React.useId();

  return (
    <div className={checked ? "ui-switch-card is-checked" : "ui-switch-card"}>
      <div className="ui-switch-copy">
        <label htmlFor={switchId} id={labelId} className="ui-switch-title">
          {title}
        </label>
        {description && (
          <span id={descId} className="ui-switch-desc">
            {description}
          </span>
        )}
      </div>
      <Switch.Root
        id={switchId}
        checked={checked}
        disabled={disabled}
        aria-labelledby={labelId}
        aria-describedby={description ? descId : undefined}
        onCheckedChange={(next) => onChange(next)}
        className="ui-switch-root"
      >
        <Switch.Thumb className="ui-switch-thumb" />
      </Switch.Root>
    </div>
  );
}

export function CompactSwitch({
  label,
  checked,
  onChange,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
}) {
  const switchId = React.useId();
  const labelId = React.useId();

  return (
    <div className="ui-compact-switch">
      <Switch.Root
        id={switchId}
        checked={checked}
        aria-labelledby={labelId}
        onCheckedChange={(next) => onChange(next)}
        className="ui-switch-root compact"
      >
        <Switch.Thumb className="ui-switch-thumb" />
      </Switch.Root>
      <label
        htmlFor={switchId}
        id={labelId}
        className="ui-compact-switch-label"
      >
        {label}
      </label>
    </div>
  );
}

function useResponsiveTabsOrientation(
  breakpoint = 980,
): "horizontal" | "vertical" {
  const [orientation, setOrientation] = useState<"horizontal" | "vertical">(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia(`(max-width: ${breakpoint}px)`).matches
        ? "horizontal"
        : "vertical",
  );

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia(`(max-width: ${breakpoint}px)`);
    const update = () =>
      setOrientation(mql.matches ? "horizontal" : "vertical");
    update();
    mql.addEventListener("change", update);
    return () => mql.removeEventListener("change", update);
  }, [breakpoint]);

  return orientation;
}

export function SettingsTabs({
  value,
  onValueChange,
  ariaLabel,
  tabs,
  children,
}: {
  value: string;
  onValueChange: (next: string) => void;
  ariaLabel: string;
  tabs: readonly {
    id: string;
    label: string;
    summary: string;
    badge?: string | null;
  }[];
  children: React.ReactNode;
}) {
  const orientation = useResponsiveTabsOrientation(980);

  return (
    <Tabs.Root
      value={value}
      orientation={orientation}
      onValueChange={(next) => {
        if (typeof next === "string") onValueChange(next);
      }}
      className="ui-tabs-root"
    >
      <Tabs.List aria-label={ariaLabel} className="ui-tabs-list">
        {tabs.map((tab) => (
          <Tabs.Tab
            key={tab.id}
            type="button"
            value={tab.id}
            className="ui-tabs-tab"
          >
            <span className="ui-tabs-tab-top">
              <span className="ui-tabs-tab-title">{tab.label}</span>
              {tab.badge !== null && tab.badge !== undefined && (
                <span className="ui-tabs-tab-badge">{tab.badge}</span>
              )}
            </span>
            <small className="ui-tabs-tab-desc">{tab.summary}</small>
          </Tabs.Tab>
        ))}
      </Tabs.List>
      <div className="ui-tabs-panels">{children}</div>
    </Tabs.Root>
  );
}

export function SettingsTabPanel({
  value,
  children,
}: {
  value: string;
  children: React.ReactNode;
}) {
  return (
    <Tabs.Panel value={value} className="ui-tabs-panel">
      {children}
    </Tabs.Panel>
  );
}

export function DisclosureGroup({
  defaultOpen = [],
  className,
  children,
}: {
  defaultOpen?: string[];
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <Accordion.Root<string>
      multiple
      defaultValue={defaultOpen}
      className={["ui-accordion", className].filter(Boolean).join(" ")}
    >
      {children}
    </Accordion.Root>
  );
}

export function DisclosureItem({
  value,
  title,
  summary,
  children,
}: {
  value: string;
  title: string;
  summary?: string;
  children: React.ReactNode;
}) {
  return (
    <Accordion.Item value={value} className="ui-accordion-item">
      <Accordion.Header className="ui-accordion-header">
        <Accordion.Trigger type="button" className="ui-accordion-trigger">
          <span className="ui-accordion-title">{title}</span>
          {summary && <span className="ui-accordion-summary">{summary}</span>}
          <IconChevronDown size={15} className="ui-accordion-chevron" />
        </Accordion.Trigger>
      </Accordion.Header>
      <Accordion.Panel className="ui-accordion-panel">
        <div className="ui-accordion-body">{children}</div>
      </Accordion.Panel>
    </Accordion.Item>
  );
}
